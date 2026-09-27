# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 ccl_dpu 动态链接库，在 host 侧使用
add_library(ccl_dpu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/communicator/hostdpu/dpu_kernel_entrance.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/communicator/hostdpu/task_service.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx/profiling/dlprof_function.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
)

# 宏定义
target_compile_definitions(ccl_dpu PRIVATE
    HCCD
    OPEN_BUILD_PROJECT
)

# 编译选项
target_compile_options(ccl_dpu PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 链接选项
target_link_options(ccl_dpu PRIVATE
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

# 链接库
target_link_libraries(ccl_dpu
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:slog_headers>
    -Wl,--no-as-needed
    c_sec
    mmpa
    runtime
    acl_rt
    -Wl,--as-needed
    -lrt
    -ldl
    -lpthread
)

# 头文件搜索路径
target_include_directories(ccl_dpu PRIVATE
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/src/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/base_comm/common

    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/types
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/common/exception
    ${CMAKE_CURRENT_SOURCE_DIR}/common/error_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/base/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/unified_platform/pub_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/unified_platform/external_system
    ${CMAKE_CURRENT_SOURCE_DIR}/service/collective/primitive
    ${CMAKE_CURRENT_SOURCE_DIR}/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx/profiling
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/host
)

# 符号隐藏: 仅导出白名单符号; UT/ST 打桩依赖动态符号, 不挂载
if(NOT ENABLE_TEST)
    target_link_options(ccl_dpu PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ccl_dpu.map")
    set_property(TARGET ccl_dpu APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccl_dpu.map")
endif()

# 指定 ccl_dpu 构建完成后安装到指定的目标位置
install(TARGETS ccl_dpu
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
install(TARGETS ccl_dpu
    LIBRARY DESTINATION ${INSTALL_DPU_KERNEL_JSON_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)

# 将 ccl_dpu.ini 转换为 json 格式
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    COMMAND ${HI_PYTHON} ${HCOMM_DIR}/cmake/scripts/parser_ini.py
                         ${CMAKE_CURRENT_LIST_DIR}/framework/communicator/hostdpu/ccl_dpu.ini
                         ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Generating libccl_dpu.json"
    VERBATIM
)
add_custom_target(ccl_dpu_json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    DESTINATION ${INSTALL_DPU_KERNEL_JSON_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
