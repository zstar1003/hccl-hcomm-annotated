# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(BASIC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)

set(TLV_DIR ${BASIC_PATH}/rdma_agent/adapter/tlv)
set(ra_adp_list
    ra_adp.c
    ra_adp_verbs.c
    ra_adp_ping.c
    ra_adp_socket.c
    ${BASIC_PATH}/rdma_agent/adapter/async/ra_adp_pool.c
    ${BASIC_PATH}/rdma_agent/adapter/async/ra_adp_async.c
    ${BASIC_PATH}/rdma_agent/adapter/ctx/ra_adp_ctx.c
    ${BASIC_PATH}/common/dl_hal_function.c
    ${BASIC_PATH}/common/ascend_hal_dl.c
    ${BASIC_PATH}/inc/private/network/log/config_log.c
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/ra_adp_tlv.c>
)

#编译目标
add_library(ra_adp SHARED
    ${ra_adp_list}
)

target_compile_definitions(ra_adp PRIVATE
    LOG_CPP
)

target_include_directories(ra_adp PRIVATE
    ${BASIC_PATH}/inc/network
    ${BASIC_PATH}/inc/private/network
    ${BASIC_PATH}/inc/private/network/log
    ${BASIC_PATH}/rdma_agent/inc
    ${BASIC_PATH}/rdma_agent/hdc/
    ${BASIC_PATH}/rdma_agent/hdc/async/
    ${BASIC_PATH}/rdma_agent/comm
    ${BASIC_PATH}/rdma_agent/adapter
    ${BASIC_PATH}/rdma_agent/adapter/async/
    ${BASIC_PATH}/rdma_service
    ${BASIC_PATH}/common
    ${TLV_DIR}
    ${BASIC_PATH}/rdma_service/tlv
    # rdma头文件
    ${BASIC_PATH}/external_depends/host_lite

    # HCCP Legacy
    ${BASIC_PATH}/rdma_agent/adapter/ctx
    ${BASIC_PATH}/rdma_service/ctx
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(ra_adp PRIVATE
        ${TOP_DIR}/driver/driver/pkg_inc/
    )
endif()

#编译选项
target_compile_options(ra_adp PRIVATE
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
    target_link_options(ra_adp PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ra_adp.map")
    set_property(TARGET ra_adp APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ra_adp.map")
endif()

target_link_options(ra_adp PRIVATE
    -Wl,-Bsymbolic
    -Wl,--no-undefined
    -rdynamic
)

#链接动态库
target_link_libraries(ra_adp PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    rs_device
    c_sec
    unified_dlog
    -Wl,--as-needed
    -lrt
    -ldl
    -pthread
)
