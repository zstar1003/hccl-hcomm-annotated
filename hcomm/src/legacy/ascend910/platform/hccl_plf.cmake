# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 hccl_plf 动态链接库，在 host 侧使用
add_library(hccl_plf SHARED)

# 宏定义
target_compile_definitions(hccl_plf PRIVATE
    $<$<STREQUAL:${PRODUCT_SIDE},device>:ASCEND_310P_DEVICE>
    USE_AICORE_REDUCESUM
    USE_AICORE_GATHERV2
    USE_AICORE_GATHERV2_INFER
)

# 编译选项
target_compile_options(hccl_plf PRIVATE
    -Werror
    -Wno-deprecated-declarations
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 头文件搜索路径
target_include_directories(hccl_plf PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/adapter/host
    ${CMAKE_CURRENT_SOURCE_DIR}/task
    ${CMAKE_CURRENT_SOURCE_DIR}/common/
    ${CMAKE_CURRENT_SOURCE_DIR}/common/buffer_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/common/unfold_cache
    ${CMAKE_CURRENT_SOURCE_DIR}/common/p2p_mgmt
    ${CMAKE_CURRENT_SOURCE_DIR}/common/unique
    ${CMAKE_CURRENT_SOURCE_DIR}/common/misc/network_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/common/misc/gradient_segment
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport/device
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport/host
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport/heterog
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport/onesided
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/transport/onesided/device
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/mem
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/dispatcher_ctx
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/notify
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/socket
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/stream
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/rma_buffer
    ${CMAKE_CURRENT_SOURCE_DIR}/resource/
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_access
    ${CMAKE_CURRENT_SOURCE_DIR}/task
    ${CMAKE_CURRENT_SOURCE_DIR}/tbe_vector_reduce
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_communication
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/base_comm/common

    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl/
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl

    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/aicpu/
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu

    # legacy (ascend950)
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_graph
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hccl_plf PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP
    )

    target_link_libraries(hccl_plf
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:hccl_legacy_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        mmpa
        runtime
        acl_rt
        error_manager
        hccl_legacy
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    PUBLIC
        hccl_headers
    )
else()
    target_include_directories(hccl_plf PRIVATE
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/abl/msprof/inc/
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/inc/driver/
        ${TOP_DIR}/metadef/pkg_inc/
        ${TOP_DIR}/metadef/inc/external/
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/open_source/json/include
 	    ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal
 	    ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal/hcomm
    )

    target_link_libraries(hccl_plf
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:hccl_headers>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        mmpa
        runtime
        acl_rt
        error_manager
        hccl_legacy
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
        ofed_headers
    PUBLIC
        hccl_headers
    )
endif()

# 安装
# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(hccl_plf PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hccl_plf.map")
    set_property(TARGET hccl_plf APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hccl_plf.map")
endif()

install(TARGETS hccl_plf
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
