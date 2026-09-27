# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(hccl_llt_stub
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_profiling_plugin.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_gdr.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_sal.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_fe.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_error_manager.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_mc2.cc
    # ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_GenRankTable.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_gert.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_platform.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rts.cc
    ${HCCL_LLT_ROOT}/stub/llt_stub_sec.c
    ${HCCL_LLT_ROOT}/stub/llt_mmpa_stub.c
    ${HCCL_LLT_ROOT}/stub/llt_stub_mm.c
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rts.cc
    ${HCCL_LLT_ROOT}/stub/legacy/interface_channel.cc
    ${HCCL_LLT_ROOT}/stub/llt_next_orion_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_next_hccp_stub.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_rank_graph.cc
    ${HCCL_LLT_ROOT}/stub/llt_next_rootinfo_stub.cc

    ${HCCL_LLT_ROOT}/stub/llt_next_orion_ccu_stub.cc

    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_orion_adapter.cc
    ${HCCL_LLT_ROOT}/stub/llt_hccl_stub_dpu_kernel_entrance.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/hostdpu/task_service.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/hostdpu/dpu_kernel_entrance.cc
)

file(GLOB_RECURSE ALGORITHM_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/*.cc")
file(GLOB_RECURSE COMMON_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/common/*.cc")
file(GLOB_RECURSE FRAMEWORK_HOST_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/*.cc")
file(GLOB_RECURSE FRAMEWORK_DEVICE_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/*.cc")
file(GLOB_RECURSE HCCD_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/hccd/*.cc")
file(GLOB_RECURSE PLATFORM_SOURCES "${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/*.cc")
file(GLOB_RECURSE BASE_COMM "${HCCL_CODE_ROOT}/src/base_comm/*.cc")
file(GLOB_RECURSE COLL_COMMUNICATOR_MGR "${HCCL_CODE_ROOT}/src/coll_communicator_mgr/*.cc")
list(REMOVE_ITEM COLL_COMMUNICATOR_MGR
    "${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder/detour_service.cc"
    "${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/host/dfx_dlprof_function.cc"
    "${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/host/dfx_profiling_reporter.cc"
)

set(EXCLUDE_FREAMEWORK_DEVICE_REGEX
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/hccl_communicator_attrs_device.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/hccl_communicator_device.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/config/env_config_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/hccl_comm_device.cc"
    "${HCCL_CODE_ROOT}/src/coll_communicator_mgr/api_c_adpt/dev/dev_coll_comm_c_adpt.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework/zero_copy_address_mgr.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_aicpu/launch_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/op_base/src/op_base_device.cc"
    )
list(REMOVE_ITEM FRAMEWORK_HOST_SOURCES ${EXCLUDE_FREAMEWORK_DEVICE_REGEX})
list(REMOVE_ITEM COLL_COMMUNICATOR_MGR ${EXCLUDE_FREAMEWORK_DEVICE_REGEX})

set(EXCLUDE_ALGORITHM_DEVICE_REGEX 
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/hccl_alg_device.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/hccl_aiv_device.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all/coll_all_to_all_executor_aicpu.cc"
    )
list(REMOVE_ITEM ALGORITHM_SOURCES ${EXCLUDE_ALGORITHM_DEVICE_REGEX})

set(EXCLUDE_COMMON_DEVICE_REGEX 
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/plugin_runner_device.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/common/stream/stream_utils_aicpu.cc"
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_device/launch_device.cc"
    )
list(REMOVE_ITEM COMMON_SOURCES ${EXCLUDE_COMMON_DEVICE_REGEX})

set(EXCLUDE_PLATFORM_DEVICE_REGEX
    "${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/mem/mem_mapping_manager_device.cc"
    )
list(REMOVE_ITEM PLATFORM_SOURCES ${EXCLUDE_PLATFORM_DEVICE_REGEX})

list(FILTER PLATFORM_SOURCES EXCLUDE REGEX ".*src/legacy/ascend910/platform/legacy/.*")

list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/resources/comm_engine_res/engine_ctxs/engine_ctx_mgr.cc")
list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/resources/comm_engine_res/notifys/.*")
list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/resources/comm_engine_res/comm_engine_res.cc")
list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/resources/comm_engine_res/comm_engine_res_mgr.cc")
list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/primitives/aiv/.*")
list(FILTER BASE_COMM EXCLUDE REGEX ".*src/base_comm/primitives/aicpu/.*")
list(FILTER BASE_COMM EXCLUDE REGEX "src/base_comm/primitives/api_c_adpt/aicpu_ts_primitives_c_adpt.cc")
list(FILTER BASE_COMM EXCLUDE REGEX "src/base_comm/primitives/api_c_adpt/aicpu_ts_sync_data_c_adpt.cc")
list(FILTER BASE_COMM EXCLUDE REGEX "src/base_comm/primitives/api_c_adpt/aicpu_ts_diag_c_adapt.cc")

list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/legacy/ascend910/framework/common/src/topo/topo_device/topoinfo_ranktableParser_device.cc")
list(FILTER FRAMEWORK_HOST_SOURCES EXCLUDE REGEX "src/legacy/ascend910/framework/common/src/topo/topo_device/.*")

set(orion_list
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/aicpu_ts_thread.cc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/thread.cc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/cpu_ts_thread.cc

    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/aicpu_ts_thread_interface.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/exception/exception_definition.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/aicpu/ub_transport_lite_impl.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/aicpu/p2p_transport_lite_impl.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/aicpu/roce_transport_lite_impl.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/buffer.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu/rma_buf_slice_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/aicpu/notify_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/rts_notify.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/ipc_remote_notify.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu/rma_buffer_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu/rma_conn_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu/rdma_conn_lite_v2.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu/rmt_rma_buf_slice_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu/ub_conn_lite_mgr.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/host_ub_connection.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/stream.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/task/task.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/rank_graph_interface.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/virtual_topo.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/env_config/base_config_legacy.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/env_config/env_config.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/env_config/env_func.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/rdma_handle_manager.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/tokenInfo_manager.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/socket/socket_handle_manager.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/hccp_hdc_manager.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/tp_qos.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/rank_consistency_checker_v2.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/checkcrc.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_a5.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu/rtsq_base.cc
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu/sqe_build_a5.cc
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache/aicpu_task_cache_manager.cc
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache/aicpu_task_cache_entry.cc
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache/aicpu_task_cache.cc
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/aicpu_task_cache_c_adpt.cc
)

set(hccl_list
    ${hccl_llt_stub}
    ${HCCD_SOURCES}
    ${COMMON_SOURCES}
    ${FRAMEWORK_HOST_SOURCES}
    ${ALGORITHM_SOURCES}
    ${PLATFORM_SOURCES}
    ${orion_list}
    ${HCCL_LLT_ROOT}/stub/llt_detour_service_compat.cc
    ${BASE_COMM}
    ${COLL_COMMUNICATOR_MGR}
)

add_library(hccl_llt SHARED  ${hccl_list})

target_compile_definitions(hccl_llt PRIVATE
    OPEN_BUILD_PROJECT
    ENABLE_EXPERIMENTAL
)

target_include_directories(hccl_llt BEFORE PRIVATE
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info
)

#头文件搜索路径
target_include_directories(hccl_llt  PRIVATE
    ${HCCL_LLT_ROOT}/stub
    ${HCCL_LLT_ROOT}/stub/legacy/

    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCCL_CODE_ROOT}/include/
    ${HCCL_CODE_ROOT}/include/hccl/
    ${HCCL_CODE_ROOT}/pkg_inc
    ${HCCL_CODE_ROOT}/pkg_inc/hccl
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm
    ${HCCL_CODE_ROOT}/pkg_inc/hcomm/ccu 
    ${HCCL_CODE_ROOT}/pkg_inc/legacy
    ${HCCL_CODE_ROOT}/pkg_inc/legacy/hccl
    ${HCCL_CODE_ROOT}/include/ccu
    ${HCCL_CODE_ROOT}/external_depends/tsch
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/inner
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc/new
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/inc/private/network
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/aubdfx
    ${HCCL_CODE_ROOT}/src/base_comm/resources/hccp/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/team/hcomm
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/team/hccl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/symmetric_memory
    ${HCCL_CODE_ROOT}/src/pkg_inc
    ${HCCL_CODE_ROOT}/external_depends

    # hccl/hccl目录下头文件
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/pub_inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/pub_inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/inc/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/inc/host/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/health/heartbeat/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/detect/detect_connect_anomalies/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/snapshot
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/cluster_maintenance/aclgraph
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/config
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/task
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/topo
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/thread
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/common/src/onesided_memory_management
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/one_sided_service
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/independent_op/channel/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/config_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads/order_launch_thread_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/launch
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/engine_ctxs
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/hcom/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/hcom/gradient_segment
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/op_base/src/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/group
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/nslbdp/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/socket/

    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/unique
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/unfold_cache
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/buffer_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc/adapter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/error_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/inc/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/host/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/heterog/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/onesided/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/transport/onesided/device
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/notify/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/rma_buffer
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/socket
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/aiv_communication/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/hccd

    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/resource/dispatcher_ctx

    # hccl/common 目录下头文件
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/health/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/config
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/debug/profiling/inc/host
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/stream/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common/launch_device/

    # hccl/algorithm 目录下头文件
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_aiv_template
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/inc_all_reduce_deter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/mc2_handler
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/legacy/operator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/task
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/operator
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/operator/registry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/registry
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_gather_v/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_reduce/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_send_receive
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_reduce_scatter_v/310P
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_all_to_all
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/impl/coll_executor/coll_scatter

    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_gather
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_all_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoall
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_alltoallv
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_broadcast
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_reduce_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/temp_scatter
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template/component
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/algorithm/base/pub_inc

    # hccl/operator目录下头文件
    ${HCCL_CODE_ROOT}/operator

    # /hccl/framework/device/目录下头文件
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc/inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/aicpu_kfc/common
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads

    # proto.h头文件
    ${CMAKE_BINARY_DIR}/proto/hccl_llt

    # tbe头文件
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/hccl_tbe_task/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/common/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/graph_ctx_mgr/
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/legacy/inc/

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

    ${JSON_INCLUDE_DIR}

    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common/

    ${HCCL_CODE_ROOT}/src/base_comm
    ${HCCL_CODE_ROOT}/src/base_comm/common/
    ${HCCL_CODE_ROOT}/src/base_comm/common/debug
    ${HCCL_CODE_ROOT}/src/base_comm/common/device
    ${HCCL_CODE_ROOT}/src/base_comm/dfx/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aiv
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/ccu
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/host
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/host/dpu_notify
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints
    ${HCCL_CODE_ROOT}/src/base_comm/resources/reged_mems
    ${HCCL_CODE_ROOT}/src/base_comm/resources/api_c_adpt
    ${HCCL_CODE_ROOT}/src/base_comm/resources/common
    ${HCCL_CODE_ROOT}/src/base_comm/resources/common/device
    ${HCCL_CODE_ROOT}/src/base_comm/config_mgr
    ${HCCL_CODE_ROOT}/src/base_comm/resources
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/stream/aicpu/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/types
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/utils
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/exception/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/external_system
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/transport/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/env_config
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/service/collective/primitive
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator/aicpu/inc
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/sockets
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/mem
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/socket/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/unified_platform/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/service/collective
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/communicator
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface

    # ascendc
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/highlevel_api/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/
    ${ASCEND_CANN_PACKAGE_PATH}/include/adv_api/hccl/internal/
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/

    #next
    ${HCCL_CODE_ROOT}/src/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads
    ${HCCL_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/config_mgr
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems/
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/common/loggers
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/api_c_adpt/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu/device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/sockets/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/dfx
    ${HCCL_CODE_ROOT}/src/base_comm/resources/reged_mems
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/api_c_adpt/

    ${HCCL_CODE_ROOT}/src/base_comm/primitives
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/aicpu/task_cache
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoints/server_socket/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/southbound_adpt/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_transport/
 
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_kernel
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_instance
    ${HCCL_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/ccu/
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_dfx
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_representation
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode_opt
    ${HCCL_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode_opt/config
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

    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/cluster_monitor
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/host
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/aicpu
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/host
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework
    ${HCCL_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/slaves
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/task_exception
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/profiling
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/dfx/
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/interface/

)

FILE(GLOB_RECURSE sources_list LIST_DIRECTORIES TRUE ${HCCL_CODE_ROOT}/src/legacy/ascend950)
foreach (dir ${sources_list})
    IF ((IS_DIRECTORY ${dir}) AND (${dir} MATCHES "src/legacy/ascend950") AND (NOT ${dir} MATCHES "src/legacy/ascend950/mdpi"))
        target_include_directories(hccl_llt  PRIVATE  ${dir})
    endif()
endforeach()

target_compile_definitions(hccl_llt    PRIVATE
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    -DCCL_LLT
)

target_compile_options(hccl_llt   PRIVATE
    -O0 -U_FORTIFY_SOURCE -g --coverage -fprofile-arcs -ftest-coverage
)

target_link_libraries(hccl_llt  PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--allow-multiple-definition
    -lgcov
    -lrt
    -ldl
)
