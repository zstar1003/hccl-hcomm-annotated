# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 hccl_alg 链接库，在 host 侧使用
add_library(hccl_alg SHARED)

# 编译选项
target_compile_options(hccl_alg PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 头文件搜索路径
target_include_directories(hccl_alg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/base/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_aiv_template
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/component/
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_alltoall
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_alltoallv
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/base/alg_template/temp_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/base/mc2_handler
    ${CMAKE_CURRENT_SOURCE_DIR}/base/communicator
    ${CMAKE_CURRENT_SOURCE_DIR}/base/communicator/legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/impl
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/legacy/operator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/resource_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/task
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/operator/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/registry
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_send_receive
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_reduce/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_to_all
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather_v
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_all_gather_v/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter_v
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_reduce_scatter_v/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_scatter
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast
    ${CMAKE_CURRENT_SOURCE_DIR}/impl/coll_executor/coll_broadcast/310P
    ${CMAKE_CURRENT_SOURCE_DIR}/pub_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/health
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/profiling/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/profiling/inc/host
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/debug/config
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/stream/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/launch_device

    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl

    # pub_inc (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc

    # framework (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc/
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config

    # platform (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/platform/task/
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter/
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common/

    # hccp (base_comm/resources)
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network/
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/
    ${HCOMM_DIR}/src/base_comm/common

    # common
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl_alg PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP
    )

    target_link_libraries(hccl_alg PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:acl_rt_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
        hccl_plf
    )
else()
    target_include_directories(hccl_alg PRIVATE
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/metadef/inc/external/
	    ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/ace/npuruntime/inc/runtime
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/runtime/include/external
    )

    target_link_libraries(hccl_alg PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        mmpa
        -Wl,--no-as-needed
        hccl_plf
        ofed_headers
    )

    install_package(
        PACKAGE hccl
        TARGETS hccl_alg
    )
endif()

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(hccl_alg PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hccl_alg.map")
    set_property(TARGET hccl_alg APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hccl_alg.map")
endif()

install(TARGETS hccl_alg
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
