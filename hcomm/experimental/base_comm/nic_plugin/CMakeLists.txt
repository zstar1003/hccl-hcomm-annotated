# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(hcomm_experimental_plugins LANGUAGES C CXX)

if(NOT DEFINED PRODUCT_SIDE)
    set(PRODUCT_SIDE "host")
endif()

if(DEFINED HCOMM_DIR)
    set(HCOMM_SOURCE_DIR "${HCOMM_DIR}")
else()
    set(HCOMM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "hcomm repository root")
endif()
set(HCOMM_EXPERIMENTAL_HCOMM_LIBRARY "" CACHE FILEPATH "libhcomm.so path used when building host NIC plugins standalone")
set(HCOMM_EXPERIMENTAL_HCCL_V2_LIBRARY "" CACHE FILEPATH "libhccl_v2.so path used when building host NIC plugins standalone")
set(HCOMM_EXPERIMENTAL_HCCL_PLF_LIBRARY "" CACHE FILEPATH "libhccl_plf.so path used when building host NIC plugins standalone")
set(HCOMM_EXPERIMENTAL_RA_LIBRARY "" CACHE FILEPATH "libra.so path used when building host NIC plugins standalone")

set(HCOMM_EXPERIMENTAL_LEGACY_INCLUDE_LIST)
file(GLOB_RECURSE HCOMM_EXPERIMENTAL_LEGACY_DIRS LIST_DIRECTORIES TRUE ${HCOMM_SOURCE_DIR}/src/legacy)
foreach(dir ${HCOMM_EXPERIMENTAL_LEGACY_DIRS})
    if(IS_DIRECTORY ${dir})
        list(APPEND HCOMM_EXPERIMENTAL_LEGACY_INCLUDE_LIST ${dir})
    endif()
endforeach()

function(hcomm_experimental_plugin_common target_name output_name)
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME ${output_name}
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(${target_name} PRIVATE
        -Werror
        -Wfloat-equal
        -Wall
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O3
        -fstack-protector-all
        $<$<CONFIG:Debug>:-g>
    )
    target_compile_definitions(${target_name} PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:_GLIBCXX_USE_CXX11_ABI=0>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:OPEN_BUILD_PROJECT>
        $<$<BOOL:${BUILD_OPEN_PROJECT}>:LOG_CPP>
    )
    target_include_directories(${target_name} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/../endpoint
        ${CMAKE_CURRENT_LIST_DIR}/../channel
        ${CMAKE_CURRENT_LIST_DIR}/../channel/dpu_notify
        ${CMAKE_CURRENT_LIST_DIR}/../comm_mem
        ${HCOMM_SOURCE_DIR}/include
        ${HCOMM_SOURCE_DIR}/include/ccu
        ${HCOMM_SOURCE_DIR}/include/hccl
        ${HCOMM_SOURCE_DIR}/pkg_inc
        ${HCOMM_SOURCE_DIR}/pkg_inc/hccl
        ${HCOMM_SOURCE_DIR}/pkg_inc/legacy
        ${HCOMM_SOURCE_DIR}/pkg_inc/legacy/hccl
        ${HCOMM_SOURCE_DIR}/pkg_inc/hcomm/ccu
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend910/platform/resource/notify
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend910/pub_inc
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/common
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/pub_inc
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/external_system
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/connection
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/notify
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/socket
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/stream
        ${HCOMM_SOURCE_DIR}/src/legacy/ascend950/unified_platform/resource/transport
        ${HCOMM_SOURCE_DIR}/src/pub_inc
        ${HCOMM_SOURCE_DIR}/src/pub_inc/inner
        ${HCOMM_SOURCE_DIR}/src/pub_inc/new
        ${HCOMM_SOURCE_DIR}/src/algorithm/pub_inc
        ${HCOMM_SOURCE_DIR}/src/framework
        ${HCOMM_SOURCE_DIR}/src/framework/inc
        ${HCOMM_SOURCE_DIR}/src/framework/common/src
        ${HCOMM_SOURCE_DIR}/src/framework/common/src/config
        ${HCOMM_SOURCE_DIR}/src/framework/common/src/mgr
        ${HCOMM_SOURCE_DIR}/src/framework/common/src/hashtable
        ${HCOMM_SOURCE_DIR}/src/framework/communicator/impl
        ${HCOMM_SOURCE_DIR}/src/framework/communicator/impl/resource_manager
        ${HCOMM_SOURCE_DIR}/src/framework/communicator/impl/one_sided_service
        ${HCOMM_SOURCE_DIR}/src/framework/op_base/src
        ${HCOMM_SOURCE_DIR}/src/base_comm/common
        ${HCOMM_SOURCE_DIR}/src/base_comm/common/device
        ${HCOMM_SOURCE_DIR}/src/base_comm/primitives/api_c_adpt
        ${HCOMM_SOURCE_DIR}/src/base_comm/primitives/api_c_adpt/nic_plugin
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/pub_inc
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/ccu_device
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/ccu_device/ccu_comp
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_pfe
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/ccu/ccu_device/ccu_comp/ccu_channel/ccu_channel_ctx_v1
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoints
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoints/server_socket
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/reged_mems
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs/channels
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs/channels/host
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs/channels/host/dpu_notify
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/endpoint_pairs/sockets
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/southbound_adpt
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/communicator
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/communicator/group_schedule_mgr
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/endpoints
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs
        ${HCOMM_SOURCE_DIR}/src/coll_communicator_mgr/dfx/profiling/aicpu
        ${HCOMM_SOURCE_DIR}/src/framework/common/src/h2d_tlv
        ${HCOMM_SOURCE_DIR}/src/common/debug/profiling/inc
        ${HCOMM_SOURCE_DIR}/src/common/debug/profiling/inc/host
        ${HCOMM_SOURCE_DIR}/src/common/debug/config
        ${HCOMM_SOURCE_DIR}/src/common/stream
        ${HCOMM_SOURCE_DIR}/src/common/launch_device
        ${HCOMM_SOURCE_DIR}/src/algorithm/base/inc
        ${HCOMM_SOURCE_DIR}/src/algorithm/base/alg_template
        ${HCOMM_SOURCE_DIR}/src/algorithm/base/communicator
        ${HCOMM_SOURCE_DIR}/src/algorithm/base/communicator/legacy
        ${HCOMM_SOURCE_DIR}/src/algorithm/impl
        ${HCOMM_SOURCE_DIR}/src/algorithm/impl/resource_manager
        ${HCOMM_SOURCE_DIR}/src/algorithm/impl/task
        ${HCOMM_SOURCE_DIR}/src/algorithm/impl/legacy
        ${HCOMM_SOURCE_DIR}/src/algorithm/impl/coll_executor
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/communicator/aicpu
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/resource_manager/socket
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/common
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/phy_topo
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/phy_topo_builder
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/rank_graph
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/rank_graph_builder
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/rank_table_info
        ${HCOMM_SOURCE_DIR}/src/legacy/framework/topo/new_topo_builder/topo_info
        ${HCOMM_SOURCE_DIR}/src/legacy/common
        ${HCOMM_SOURCE_DIR}/src/legacy/common/exception
        ${HCOMM_SOURCE_DIR}/src/legacy/common/types
        ${HCOMM_SOURCE_DIR}/src/legacy/common/utils
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/common
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/pub_inc
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/pub_inc/resource/transport
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/external_system
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/resource
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/resource/buffer
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/resource/connection
        ${HCOMM_SOURCE_DIR}/src/legacy/unified_platform/resource/socket
        ${HCOMM_SOURCE_DIR}/src/platform/common
        ${HCOMM_SOURCE_DIR}/src/platform/inc
        ${HCOMM_SOURCE_DIR}/src/platform/inc/adapter
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/hccp
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/hccp/inc
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/hccp/inc/network
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/hccp/inc/private/network
        ${HCOMM_SOURCE_DIR}/src/base_comm/resources/hccp/external_depends/ubengine
        ${HCOMM_SOURCE_DIR}/src/platform/resource/notify
        ${HCOMM_SOURCE_DIR}/src/platform/resource/dispatcher_ctx
        ${HCOMM_SOURCE_DIR}/src/platform/resource/socket
        ${HCOMM_SOURCE_DIR}/src/platform/resource/transport
        ${HCOMM_SOURCE_DIR}/src/platform/resource/transport/heterog
        ${HCOMM_SOURCE_DIR}/src/platform/resource/transport/host
        ${HCOMM_SOURCE_DIR}/src/platform/resource/transport/onesided
        ${HCOMM_SOURCE_DIR}/src/platform/typical
        ${HCOMM_EXPERIMENTAL_LEGACY_INCLUDE_LIST}
        ${JSON_INCLUDE_DIR}
        ${THIRD_PARTY_NLOHMANN_PATH}
        ${HCOMM_SOURCE_DIR}/third_party/json/include
        ${RDMA_CORE_INCLUDE_DIR}
        ${URMA_INCLUDE_DIR}
        ${HCOMM_SOURCE_DIR}/third_party/rdma-core-build/include
    )
    target_link_libraries(${target_name} PRIVATE
        $<$<TARGET_EXISTS:c_sec_headers>:c_sec_headers>
        $<$<TARGET_EXISTS:slog_headers>:slog_headers>
        $<$<TARGET_EXISTS:runtime_headers>:runtime_headers>
        $<$<TARGET_EXISTS:acl_rt_headers>:acl_rt_headers>
        $<$<TARGET_EXISTS:ascend_hal_headers>:ascend_hal_headers>
        $<$<TARGET_EXISTS:rdma_core_headers>:rdma_core_headers>
        -lrt
        -ldl
        -lpthread
    )
    if(NOT TARGET c_sec_headers AND DEFINED ASCEND_CANN_PACKAGE_PATH)
        target_include_directories(${target_name} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/include
        )
    endif()
    if(NOT TARGET slog_headers AND DEFINED ASCEND_CANN_PACKAGE_PATH)
        target_include_directories(${target_name} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        )
    endif()
    if(NOT TARGET runtime_headers AND DEFINED ASCEND_CANN_PACKAGE_PATH)
        target_include_directories(${target_name} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
            ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/aicpu_schedule
        )
    endif()
    if(NOT TARGET acl_rt_headers AND DEFINED ASCEND_CANN_PACKAGE_PATH)
        target_include_directories(${target_name} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/include/acl
        )
    endif()
    if(NOT TARGET ascend_hal_headers AND DEFINED ASCEND_CANN_PACKAGE_PATH)
        target_include_directories(${target_name} PRIVATE
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/driver
            ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc/driver
        )
    endif()
endfunction()

function(hcomm_experimental_plugin_link_dependency target_name dependency_target dependency_library)
    if(TARGET ${dependency_target})
        target_link_libraries(${target_name} PRIVATE ${dependency_target})
    elseif(DEFINED ${dependency_library} AND NOT "${${dependency_library}}" STREQUAL "")
        target_link_libraries(${target_name} PRIVATE ${${dependency_library}})
    else()
        message(WARNING
            "${target_name} is not linked to ${dependency_target}. "
            "Configure from the hcomm top-level project or set ${dependency_library} "
            "to catch unresolved symbols at link time."
        )
    endif()
endfunction()

function(hcomm_experimental_plugin_link_dependencies target_name)
    foreach(dependency IN LISTS ARGN)
        if(dependency STREQUAL "hcomm")
            hcomm_experimental_plugin_link_dependency(${target_name} hcomm HCOMM_EXPERIMENTAL_HCOMM_LIBRARY)
        elseif(dependency STREQUAL "hccl_v2")
            hcomm_experimental_plugin_link_dependency(${target_name} hccl_v2 HCOMM_EXPERIMENTAL_HCCL_V2_LIBRARY)
        elseif(dependency STREQUAL "hccl_plf")
            hcomm_experimental_plugin_link_dependency(${target_name} hccl_plf HCOMM_EXPERIMENTAL_HCCL_PLF_LIBRARY)
        elseif(dependency STREQUAL "ra")
            hcomm_experimental_plugin_link_dependency(${target_name} ra HCOMM_EXPERIMENTAL_RA_LIBRARY)
        else()
            message(FATAL_ERROR "Unknown host NIC plugin dependency: ${dependency}")
        endif()
    endforeach()
    target_link_options(${target_name} PRIVATE -Wl,--no-undefined)
endfunction()

add_library(hcomm_cpu_roce_plugin SHARED
    plugin_core.cc
    host_roce_plugin.cc
    ../endpoint/cpu_roce_endpoint.cc
    ../comm_mem/roce_mem.cc
    ../channel/host_cpu_roce_channel.cc
    ../channel/host_rdma_connection.cc
    ../channel/dpu_notify/dpu_notify_manager.cc
)
hcomm_experimental_plugin_common(hcomm_cpu_roce_plugin hcomm_cpu_roce_plugin)
hcomm_experimental_plugin_link_dependencies(hcomm_cpu_roce_plugin hcomm hccl_v2 hccl_plf ra)
if(TARGET json)
    add_dependencies(hcomm_cpu_roce_plugin json)
endif()

add_library(hcomm_cpu_ub_plugin SHARED
    plugin_core.cc
    host_ub_plugin.cc
    ../endpoint/cpu_urma_endpoint.cc
    ../comm_mem/urma_mem.cc
    ../channel/host_cpu_urma_channel.cc
)
hcomm_experimental_plugin_common(hcomm_cpu_ub_plugin hcomm_cpu_ub_plugin)
hcomm_experimental_plugin_link_dependencies(hcomm_cpu_ub_plugin hcomm hccl_v2 ra)
if(TARGET json)
    add_dependencies(hcomm_cpu_ub_plugin json)
endif()

# 符号隐藏: 仅导出 nic_plugin.map 的 3 个 dlsym 入口; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(hcomm_cpu_roce_plugin PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/nic_plugin.map")
    set_property(TARGET hcomm_cpu_roce_plugin APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/nic_plugin.map")
    target_link_options(hcomm_cpu_ub_plugin PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/nic_plugin.map")
    set_property(TARGET hcomm_cpu_ub_plugin APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/nic_plugin.map")
endif()

install(TARGETS hcomm_cpu_roce_plugin hcomm_cpu_ub_plugin
    LIBRARY DESTINATION hcomm_plugin ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
