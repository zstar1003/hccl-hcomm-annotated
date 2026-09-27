# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 aicpu_custom 库，在 device 侧使用
add_library(aicpu_custom SHARED)

# 克隆 ccl_kernel 属性
clone_cann_target(
    ORIGIN ccl_kernel
    OUTPUT aicpu_custom
    IGNORE_PROP LINK_LIBRARIES      # 忽略链接库属性
                    LINK_OPTIONS     # 忽略链接选项（含 ccl_kernel 的 version script, aicpu_custom 挂载独立白名单）
)

# 链接库
target_link_libraries(aicpu_custom PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:acl_rt_headers>
    $<BUILD_INTERFACE:asc_host_headers>
    $<BUILD_INTERFACE:asc_kernel_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:kernel_tiling_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:json>
    -Wl,--no-as-needed
    ascend_hal
    c_sec
    mmpa
    -Wl,--whole-archive
    ccl_kernel_plf_a            # 链接 ccl_kernel_plf 静态库
    -Wl,--no-whole-archive
    -Wl,--as-needed
    -lrt
    -ldl
    -lpthread
)

# 将 aicpu_custom.ini 转换为 json 格式
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
    COMMAND ${HI_PYTHON} ${HCOMM_DIR}/cmake/scripts/parser_ini.py
                         ${CMAKE_CURRENT_LIST_DIR}/device/framework/aicpu_custom.ini
                         ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Generating libaicpu_custom.json"
 	VERBATIM
)
add_custom_target(aicpu_custom_json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(aicpu_custom PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/aicpu_custom.map")
    set_property(TARGET aicpu_custom APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/aicpu_custom.map")
endif()

# 安装
install(TARGETS aicpu_custom
    LIBRARY DESTINATION ${INSTALL_CCL_KERNEL_JSON_DIR}/kernel ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libaicpu_custom.json
    DESTINATION ${INSTALL_CCL_KERNEL_JSON_DIR}/kernel ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
