# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(ra_list
    async/ra_comm.c
    async/ra_socket.c
    async/ra_ctx.c
    ra_nda.c
    ra_ctx.c
    ra_host.c
    ra_host_verbs.c
    ra_common.c
    ra_ping.c
    ra_rdma.c
    ra_init.c
    ra_socket.c
    ra_tlv.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log/config_log.c
)

add_library(ra  SHARED ${ra_list})

target_include_directories(ra PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../hdc
    ${CMAKE_CURRENT_SOURCE_DIR}/../hdc/async
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log
    ${CMAKE_CURRENT_SOURCE_DIR}/../comm
    ${CMAKE_CURRENT_SOURCE_DIR}/../peer
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/

    # rdma头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external_depends/host_lite

    # HCCP Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/async
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(ra PRIVATE
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/driver/driver/pkg_inc/
    )
endif()

target_compile_options(ra PRIVATE
    -Werror
    -fvisibility=hidden
    -fsigned-char
    -fno-common
    -fno-strict-aliasing
    -std=gnu11
    $<$<CONFIG:Debug>:-Og -g>
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ra PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ra.map")
    set_property(TARGET ra APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ra.map")
endif()

target_link_libraries(ra PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    ra_hdc
    ra_peer
    -Wl,--as-needed
    -ldl -lrt
)

target_compile_definitions(ra PRIVATE
    LOG_HOST
    LOG_CPP
)

# 安装
if((NOT BUILD_OPEN_PROJECT) OR (PRODUCT_SIDE STREQUAL "host"))
    install(TARGETS ra
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
