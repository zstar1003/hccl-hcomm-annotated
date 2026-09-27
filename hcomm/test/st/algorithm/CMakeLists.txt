# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(HCOMM_ALG_TC_ROOT    ${CMAKE_CURRENT_SOURCE_DIR}/testcase)
set(HCOMM_ALG_UTILS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/utils)

set(HCOMM_CODE_ROOT      ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(HCOMM_CODE_COMMON    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/common)
set(HCOMM_CODE_PLATFORM  ${HCOMM_CODE_ROOT}/src/legacy/ascend910/platform)
set(HCOMM_CODE_ALGORITHM ${HCOMM_CODE_ROOT}/src/legacy/ascend910/algorithm)
set(HCOMM_CODE_FRAMEWORK ${HCOMM_CODE_ROOT}/src/legacy/ascend910/framework)
set(HCOMM_CODE_LEGACY    ${HCOMM_CODE_ROOT}/src/legacy/ascend950)

# Include List
set(ASCEND_CANN_INCLUDE_LIST
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/driver
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/mmpa
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/asc/impl/basic_api
)

set(UTILS_PUBLIC_INCLUDE_LIST
    ${HCOMM_ALG_UTILS_ROOT}/inc
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/inc
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/checker
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/adapter_v1/aiv
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/adapter_v1/notify
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/adapter_v1/stream
    ${HCOMM_ALG_UTILS_ROOT}/pub_inc/adapter_v1/transport
)

set(HCOMM_PUBLIC_INCLUDE_LIST
    ${HCOMM_CODE_ROOT}
    ${HCOMM_CODE_ROOT}/include
    ${HCOMM_CODE_ROOT}/include/hccl
    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc
    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_CODE_ROOT}/pub_inc
    ${HCOMM_CODE_ROOT}/pub_inc/hccl
    ${HCOMM_CODE_ROOT}/pkg_inc
    ${HCOMM_CODE_ROOT}/pkg_inc/hccl
    ${HCOMM_CODE_ROOT}/pkg_inc/hcomm
    ${HCOMM_CODE_ROOT}/pkg_inc/legacy
    ${HCOMM_CODE_ROOT}/pkg_inc/legacy/hccl
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
)

set(HCOMM_PLATFORM_INCLUDE_LIST
    ${HCOMM_CODE_PLATFORM}
    ${HCOMM_CODE_PLATFORM}/inc
    ${HCOMM_CODE_PLATFORM}/inc/adapter
    ${HCOMM_CODE_PLATFORM}/inc/adapter/host
    ${HCOMM_CODE_PLATFORM}/common
    ${HCOMM_CODE_PLATFORM}/task/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/inc/log
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network/
)

set(HCOMM_COMMON_INCLUDE_LIST
    ${HCOMM_CODE_COMMON}/stream
    ${HCOMM_CODE_COMMON}/health
    ${HCOMM_CODE_COMMON}/launch_device
    ${HCOMM_CODE_COMMON}/debug/profiling
    ${HCOMM_CODE_COMMON}/debug/profiling/inc
    ${HCOMM_CODE_COMMON}/debug/profiling/inc/host
    ${HCOMM_CODE_COMMON}/debug/config
    ${HCOMM_CODE_COMMON}/debug/launch_device
)

set(HCOMM_ALGORITHM_INCLUDE_LIST
    ${HCOMM_CODE_ALGORITHM}/pub_inc
    ${HCOMM_CODE_ALGORITHM}/base
    ${HCOMM_CODE_ALGORITHM}/base/inc
    ${HCOMM_CODE_ALGORITHM}/base/communicator
    ${HCOMM_CODE_ALGORITHM}/base/communicator/legacy
    ${HCOMM_CODE_ALGORITHM}/base/alg_template
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/component
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_all_gather
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_all_reduce
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_alltoall
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_alltoallv
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_reduce_scatter
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_reduce
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_scatter
    ${HCOMM_CODE_ALGORITHM}/base/alg_template/temp_broadcast
    ${HCOMM_CODE_ALGORITHM}/utils/inc
    ${HCOMM_CODE_ALGORITHM}/impl
    ${HCOMM_CODE_ALGORITHM}/impl/inc
    ${HCOMM_CODE_ALGORITHM}/impl/task
    ${HCOMM_CODE_ALGORITHM}/impl/legacy
    ${HCOMM_CODE_ALGORITHM}/impl/operator
    ${HCOMM_CODE_ALGORITHM}/impl/operator/registry
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/registry
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_gather
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_gather_v
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_reduce
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_all_to_all
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_broadcast
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_reduce_scatter_v
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_scatter
    ${HCOMM_CODE_ALGORITHM}/impl/coll_executor/coll_send_receive
    ${HCOMM_CODE_ALGORITHM}/impl/resource_manager
)

set(HCOMM_FRAMEWORK_INCLUDE_LIST
    ${HCOMM_CODE_FRAMEWORK}
    ${HCOMM_CODE_FRAMEWORK}/inc
    ${HCOMM_CODE_FRAMEWORK}/inc/host
    ${HCOMM_CODE_FRAMEWORK}/common/src
    ${HCOMM_CODE_FRAMEWORK}/common/src/mgr
    ${HCOMM_CODE_FRAMEWORK}/common/src/task
    ${HCOMM_CODE_FRAMEWORK}/common/src/topo
    ${HCOMM_CODE_FRAMEWORK}/common/src/config
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl/resource_manager
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl/independent_op
    ${HCOMM_CODE_FRAMEWORK}/communicator/impl/independent_op/channel

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/comm_engine_res/engine_ctxs
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/comm_engine_res/launch
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/comm_engine_res/threads/device/
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/common
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator/device
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/local/my_rank/comm_engine_reses
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/resource_mgr/remote
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/communicator/device/c_adpt
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/profiling/host
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/aicpu
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/dfx/taskException/host
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph
    ${HCOMM_CODE_FRAMEWORK}/device/framework
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/slaves
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/channels/aicpu/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/sockets/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoints/
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoints/dfx
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/reged_mems/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/endpoint_pairs/
    ${HCOMM_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/
    ${HCOMM_CODE_ROOT}/src/base_comm/primitives/api_c_adpt/nic_plugin/
    ${HCOMM_CODE_ROOT}/src/base_comm/common/
    ${HCOMM_CODE_ROOT}/src/base_comm/common/device
    ${HCOMM_CODE_ROOT}/src/legacy/ascend910/common/stream
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network

    # ccu 头文件
    ${HCOMM_CODE_ROOT}/pkg_inc/hcomm/ccu/
    ${HCOMM_CODE_ROOT}/include/ccu
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/southbound_adpt/
    ${HCOMM_CODE_ROOT}/src/base_comm/common/

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_transport/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_kernel/
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_instance/

    ${HCOMM_DIR}/src/legacy/ascend910/common
)

set(HCOMM_LEGACY_INCLUDE_LIST
    ${HCOMM_CODE_LEGACY}
    ${HCOMM_CODE_LEGACY}/interface
    ${HCOMM_CODE_LEGACY}/common
    ${HCOMM_CODE_LEGACY}/common/utils
    ${HCOMM_CODE_LEGACY}/common/types
    ${HCOMM_CODE_LEGACY}/common/exception
    ${HCOMM_CODE_LEGACY}/framework/env_config
    ${HCOMM_CODE_LEGACY}/framework/misc/json_parser
    ${HCOMM_CODE_LEGACY}/framework/communicator
    ${HCOMM_CODE_LEGACY}/framework/communicator/aicpu
    ${HCOMM_CODE_LEGACY}/framework/resource_manager/socket
    ${HCOMM_CODE_LEGACY}/framework/topo
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_graph_builder
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo
    ${HCOMM_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info
    ${HCOMM_CODE_LEGACY}/service/collective
    ${HCOMM_CODE_LEGACY}/service/collective/primitive
    ${HCOMM_CODE_LEGACY}/unified_platform
    ${HCOMM_CODE_LEGACY}/unified_platform/external_system
    ${HCOMM_CODE_LEGACY}/unified_platform/common
    ${HCOMM_CODE_LEGACY}/unified_platform/ccu/
    ${HCOMM_CODE_LEGACY}/unified_platform/ccu/ccu_device/
    ${HCOMM_CODE_LEGACY}/unified_platform/ccu/ccu_device/ccu_component/
    ${HCOMM_CODE_LEGACY}/unified_platform/ccu/ccu_device/ccu_component/ccu_channel/
    ${HCOMM_CODE_LEGACY}/unified_platform/ccu/ccu_context/
    ${HCOMM_CODE_LEGACY}/unified_platform/pub_inc
    ${HCOMM_CODE_LEGACY}/unified_platform/pub_inc/ccu/
    ${HCOMM_CODE_LEGACY}/unified_platform/resource
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/transport
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/transport/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/stream/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/notify/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/socket
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/stream/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/connection/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/buffer/aicpu
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/mem
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/connection
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/task
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/notify
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/stream
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/buffer
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/channel/dpu_notify
    ${HCOMM_CODE_LEGACY}/unified_platform/resource/channel
    ${HCOMM_CODE_LEGACY}/framework/dfx/aicpu/task_exception
 	${HCOMM_CODE_LEGACY}/framework/dfx/aicpu/profiling
    ${HCOMM_CODE_LEGACY}/framework/dfx/aicpu/common
    ${HCOMM_CODE_LEGACY}/framework/dfx/aicpu
    ${HCOMM_CODE_LEGACY}/framework/dfx/task_exception
    ${HCOMM_CODE_LEGACY}/framework/dfx/profiling
    ${HCOMM_CODE_LEGACY}/framework/dfx/common
    ${HCOMM_CODE_LEGACY}/framework/dfx
    ${HCOMM_CODE_LEGACY}/framework/resource_manager/stream

    ${HCCL_CODE_ROOT}/src/common

)

add_subdirectory(testcase)
add_subdirectory(utils)
