# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../comm)
set(ra_peer_list
    ra_peer.c
    ra_peer_nda.c
    ra_peer_socket.c
    ra_peer_ctx.c
    ${BASE_DIR}/ra_comm.c
    ${BASE_DIR}/ctx/ra_ctx_comm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log/config_log.c
)

#编译目标
add_library(ra_peer SHARED ${ra_peer_list})

target_include_directories(ra_peer PRIVATE
    ${BASE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rdma_service
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/

    # rdma头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external_depends/host_lite
    ${IBV_EXTEND_INCLUDE_DIR}

    # HCCP Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rdma_agent/comm/ctx
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rdma_service/ctx
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(ra_peer PRIVATE
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/driver/driver/pkg_inc/
        ${TOP_DIR}/driver/driver/src/custom/nda/ibv_extend/
    )
endif()

target_compile_options(ra_peer PRIVATE
    -Werror
    -fno-common
    -std=gnu11
    $<$<CONFIG:Debug>:-Og -g>
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ra_peer PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ra_peer.map")
    set_property(TARGET ra_peer APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ra_peer.map")
endif()

target_link_libraries(ra_peer PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    rs
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    -Wl,--as-needed
    -ldl
)

target_compile_definitions(ra_peer PRIVATE
    LOG_HOST
    LOG_CPP
)

# 安装
if((NOT BUILD_OPEN_PROJECT) OR (PRODUCT_SIDE STREQUAL "host"))
    install(TARGETS ra_peer
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
