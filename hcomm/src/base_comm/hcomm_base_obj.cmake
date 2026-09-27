# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hcomm_base_obj OBJECT)

# 编译选项（与hcomm库保持一致）
target_compile_options(hcomm_base_obj PRIVATE
    -Werror
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -fno-common
    -fno-strict-aliasing
    -fPIC
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)



target_compile_definitions(hcomm_base_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>
)


# 编译定义（与hcomm库保持一致）
if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hcomm_base_obj PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP
    )
endif()

# 设置头文件搜索路径（基于依赖分析显式列出）
target_include_directories(hcomm_base_obj PRIVATE
    # ============================================================
    # 1. 对外接口头文件
    # ============================================================
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/include/ccu
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/hcomm
    ${HCOMM_DIR}/pkg_inc/hcomm/ccu
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl

    # ============================================================
    # 2. base_comm 内部目录
    # ============================================================
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device/ccu_comp
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device/ccu_comp/ccu_channel
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v2
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_dfx
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_instance
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_microcode
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/context
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/interface
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/arithmetic
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/common
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/control
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/data
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/loop
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/sync
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_representation/reps/translator
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/ccu_transport
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/ccu/pub_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/hccp/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/hccp/inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/hccp/inc/private
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/hccp/inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/hccp/external_depends/ubengine


    # ============================================================
    # 3. legacy/ascend910 目录
    # ============================================================
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_device
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/hcom
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/mgr
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/host
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/op_base/src
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/dispatcher_ctx
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/socket

    # ============================================================
    # 4. legacy/ascend950 目录
    # ============================================================
    ${HCOMM_DIR}/src/legacy/ascend950
    ${HCOMM_DIR}/src/legacy/ascend950/framework
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/profiling
    ${HCOMM_DIR}/src/legacy/ascend950/framework/env_config
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/common
    ${HCOMM_DIR}/src/legacy/ascend950/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/primitive
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_context
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_pfe
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device/ccu_component/ccu_channel
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/dfx
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_microcode
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/context
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/interface
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/arithmetic
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/control
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/data
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/loop
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/sync
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/translator
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/ccu_transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/mem
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/task
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport/aicpu

    # ============================================================
    # 5. coll_communicator_mgr 目录 (临时依赖，后续消除)
    # ============================================================
    ${HCOMM_DIR}/src/coll_communicator_mgr/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator/device
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/aicpu
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/host
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs

    # ============================================================
    # 6. 三方件头文件
    # ============================================================
    ${URMA_INCLUDE_DIR}
)

# base_comm根目录源文件
target_sources(hcomm_base_obj PRIVATE
    hcomm_res_mgr.cc
)

# 链接CANN库以获取必要的头文件路径
target_link_libraries(hcomm_base_obj
    $<BUILD_INTERFACE:c_sec_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:json>
    $<BUILD_INTERFACE:error_manager_headers>
    $<BUILD_INTERFACE:acl_rt_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:hccl_legacy_headers>
    $<BUILD_INTERFACE:asc_host_headers>
    $<BUILD_INTERFACE:kernel_tiling_headers>
)
