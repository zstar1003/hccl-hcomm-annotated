# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 hccp_service.bin 可执行文件，在 Device 侧使用
add_executable(hccp_service.bin
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_service/main.c
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_service/param.c
    ${CMAKE_CURRENT_SOURCE_DIR}/common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/private/network/log/config_log.c
)

target_include_directories(hccp_service.bin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_service
    ${CMAKE_CURRENT_SOURCE_DIR}/external_depends/host_lite
    ${CMAKE_CURRENT_SOURCE_DIR}/external_depends/rdma-core/providers/hns
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/private/network/log
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_service
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_agent/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_agent/adapter/async
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_agent/hdc
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_agent/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

target_compile_definitions(hccp_service.bin PRIVATE
    google=ascend_private
    LOG_CPP
)

target_compile_options(hccp_service.bin PRIVATE
    -Werror
    -fPIE
    -rdynamic
    -fvisibility=hidden
    -fsigned-char
    -fno-common
    -fno-strict-aliasing
    -std=gnu11
    $<$<CONFIG:Debug>:-Og -g>
)

target_link_options(hccp_service.bin PRIVATE
    -Wl,-Bsymbolic
    -Wl,--no-undefined
    -rdynamic
)

target_link_libraries(hccp_service.bin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    c_sec
    unified_dlog
    ra_adp
    rs_device
    mmpa
    tsd_eventclient
    -ldl
)
