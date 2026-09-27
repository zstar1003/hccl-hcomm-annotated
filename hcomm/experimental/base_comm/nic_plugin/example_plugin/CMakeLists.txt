# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 独立编译示例：插件仅依赖 HCOMM 的 SDK ABI 头文件 hcomm_nic_plugin.h，不链接 libhcomm.so。
# hcomm_nic_plugin.h 传递依赖 include/ 下的公开头文件（hcomm_primitives.h / hcomm_res.h 等）
# 与 securec.h / acl/acl_rt.h。

cmake_minimum_required(VERSION 3.16)

project(hcomm_example_plugin LANGUAGES CXX)

if(NOT DEFINED HCOMM_DIR)
    set(HCOMM_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../.." CACHE PATH "hcomm repository root")
endif()

# securec.h / acl/acl_rt.h：真实部署时来自 CANN 软件包；仓库内示例回退到 test stub。
if(NOT DEFINED HCOMM_EXAMPLE_SECUREC_ACL_INCLUDE_DIR)
    set(HCOMM_EXAMPLE_SECUREC_ACL_INCLUDE_DIR "${HCOMM_DIR}/test/stub/depends/include"
        CACHE PATH "securec.h / acl/acl_rt.h include dir")
endif()

add_library(example_plugin SHARED example_plugin.cc)

set_target_properties(example_plugin PROPERTIES
    OUTPUT_NAME example_plugin
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(example_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCOMM_EXAMPLE_SECUREC_ACL_INCLUDE_DIR}
)

target_compile_options(example_plugin PRIVATE -Wall -Wextra -O2)

target_link_libraries(example_plugin PRIVATE -ldl -lpthread -lrt)

# 仅导出 3 个 dlsym 入口，其余符号隐藏
target_link_options(example_plugin PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/example_plugin.map")
set_property(TARGET example_plugin APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/example_plugin.map")
