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
set(ASYNC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/async)

set(ra_hdc_list 
    ra_hdc.c
    ${BASE_DIR}/ra_comm.c
    ${BASE_DIR}/ctx/ra_ctx_comm.c
    async/ra_hdc_async_ctx.c
    ra_hdc_ctx.c
    ra_rdma_lite.c
    ra_hdc_lite.c
    ra_hdc_ping.c
    ra_hdc_socket.c
    ra_hdc_rdma.c
    ra_hdc_tlv.c
    ${ASYNC_DIR}/ra_hdc_async.c
    ${ASYNC_DIR}/ra_hdc_async_socket.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log/config_log.c
)

add_library(ra_hdc SHARED ${ra_hdc_list})

target_include_directories(ra_hdc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private
    ${CMAKE_CURRENT_SOURCE_DIR}/../../rdma_service
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/private/network/log
    ${BASE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${ASYNC_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common/

    # rdma头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external_depends/rdma-core/providers/hns
    ${CMAKE_CURRENT_SOURCE_DIR}/../../external_depends/host_lite

    # HCCP Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/../comm/ctx
    ${CMAKE_CURRENT_SOURCE_DIR}/../hdc
    ${CMAKE_CURRENT_SOURCE_DIR}/../hdc/async
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(ra_hdc PRIVATE
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/driver/driver/pkg_inc/
    )
endif()

target_compile_options(ra_hdc PRIVATE
    -Werror
    -fsigned-char
    -fno-common
    -fno-strict-aliasing
    -std=gnu11
    $<$<CONFIG:Debug>:-Og -g>
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ra_hdc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ra_hdc.map")
    set_property(TARGET ra_hdc APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ra_hdc.map")
endif()

target_link_libraries(ra_hdc PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    -Wl,--as-needed
    -lrt
    -ldl
)

target_compile_definitions(ra_hdc PRIVATE
    LOG_HOST
    LOG_CPP
)

# 安装
if((NOT BUILD_OPEN_PROJECT) OR (PRODUCT_SIDE STREQUAL "host"))
    install(TARGETS ra_hdc
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
