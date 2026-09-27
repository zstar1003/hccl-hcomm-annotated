# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(TLV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tlv)

set(RS_SOURCE_LIST
    rs.c
    rs_epoll.c
    rs_ping.c
    rs_ping_inner.c
    rs_ping_roce.c
    rs_socket.c
    rs_rdma.c
    rs_nda.c
    rs_comm.c
    rs_drv_sensor.c
    rs_drv_rdma.c
    rs_drv_socket.c
    dl_ibverbs_function.c
    dl_ibv_extend_function.c
    hccp_dl.c
    ctx/dl_net_function.c
    ctx/dl_ccu_function.c
    ctx/dl_urma_function.c
    ctx/dl_aubdfx_function.c
    ctx/rs_ping_urma.c
    ctx/rs_ctx.c
    ctx/rs_esched.c
    ctx/rs_ub.c
    ctx/rs_ub_dfx.c
    ctx/rs_ub_tp.c
    ctx/rs_ub_jetty.c
    ctx/rs_ub_jfc.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/dl_hal_function.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/ascend_hal_dl.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/file_opt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/network_comm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/private/network/log/config_log.c
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/dl_netco_function.c>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/rs_tlv.c>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:${TLV_DIR}/rs_adp_nslb.c>
)

# 创建 rs 动态链接库，host 和 device 侧使用，同时打包到 cann-hccd-compat.tar.gz
add_library(rs SHARED ${RS_SOURCE_LIST})

# 编译选项
target_compile_options(rs PRIVATE
    -Werror
    -Wno-unused-parameter
    -fvisibility=hidden
    -fsigned-char
    -fno-common
    -fno-strict-aliasing
    -std=gnu11
    $<$<CONFIG:Debug>:-Og -g>
)

# 链接选项
target_link_options(rs PRIVATE
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

# 宏定义
target_compile_definitions(rs PRIVATE
    LOG_CPP
    $<$<STREQUAL:${PRODUCT_SIDE},host>:LOG_HOST>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:google=ascend_private>
    $<$<STREQUAL:${PRODUCT_SIDE},device>:CONFIG_TLV>
)

# 链接库
target_link_libraries(rs PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:msprof_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    -Wl,--no-as-needed
    c_sec
    $<$<STREQUAL:${PRODUCT_SIDE},host>:unified_dlog>
    tls_adp
    -Wl,--as-needed
    -Wl,--whole-archive
    -Wl,--no-whole-archive
    -lrt
    -ldl
)

# 头文件搜索路径
target_include_directories(rs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/private/network/log
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/
    ${TLV_DIR}
    # 第三方库依赖路径
    ${OPENSSL_INCLUDE_DIR}
    # rdma头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/host_lite
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/rdma-core/providers/hns
    # HCCP Legacy
    ${CMAKE_CURRENT_SOURCE_DIR}/ctx
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/net_adapt
    ${URMA_INCLUDE_DIR}
    ${IBV_EXTEND_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../external_depends/aubdfx
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(rs PRIVATE
        ${TOP_DIR}/ubengine/ssapi/userspace/umdk/src/urma/lib/urma/core/include
        ${TOP_DIR}/ubengine/ssapi/userspace/umdk/src/urma/lib/urma/core
        ${TOP_DIR}/ubengine/ssapi/userspace/udma/src/urma/hw/udma/include
        ${TOP_DIR}/drivers/network/tls_adp
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/driver/driver/pkg_inc/
        ${TOP_DIR}/driver/driver/src/custom/nda/ibv_extend/
    )
endif()

# 设置依赖
add_dependencies(rs crypto_static)

# 创建 rs 动态链接库，device 侧使用，同时打包到 cann-hcomm-compat.tar.gz
# 注意：rs_device 与 rs 完全一致，区别是编译时使能 CUSTOM_INTERFACE 宏定义
if(PRODUCT_SIDE STREQUAL "device")
    add_library(rs_device SHARED)
    # 克隆 rs 的属性
    clone_cann_target(
        ORIGIN rs
        OUTPUT rs_device
    )
    # 宏定义
    target_compile_definitions(rs_device PRIVATE
        CUSTOM_INTERFACE
    )
    # 设置依赖
    add_dependencies(rs_device crypto_static)
    # 设置 rs_device 输出文件名，同时避免与 rs 输出目录冲突
    set_target_properties(rs_device
        PROPERTIES
        OUTPUT_NAME rs
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/device"
    )
    # 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
    if(NOT ENABLE_TEST)
        target_link_options(rs_device PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/rs_device.map")
        set_property(TARGET rs_device APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/rs_device.map")
    endif()
endif()

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(rs PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/rs.map")
    set_property(TARGET rs APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/rs.map")
endif()

# 安装
if((NOT BUILD_OPEN_PROJECT) OR (PRODUCT_SIDE STREQUAL "host"))
    install(TARGETS rs
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
